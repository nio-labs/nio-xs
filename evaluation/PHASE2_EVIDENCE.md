# Phase 2 evidence — 0.1.0-alpha.2

Trained non-LLM multinomial Naive Bayes classifier, Python offline training, Rust inference and typed workflow routing. Default model remains nio-nano. No Python, NioJS, Chromium, LLM, or search API is needed by the provider runtime.

## Model and evaluation

[Training script](../training/train.py) produces a 52,228-byte JSON model with 7 labels, vocabulary, log weights, feature/schema versions, dataset digest, and validation-selected abstention threshold 0.8. Embedded artifacts are validated at startup. External model loading requires an operator-supplied SHA-256 and the same validated schema. Rust matches Python probabilities within 1e-10 on all 300 development fixtures. Invalid metadata/weight shapes fail startup.

The original 1,270-record synthetic corpus has 770 training, 200 validation, and 300 development-test records with family-separated splits. The development test was inspected while improving feature design; it is not presented as untouched acceptance data. The initial poor result is retained in phase2-initial-development.json.

A separate [post-freeze corpus](phase2-frozen-holdout.jsonl) contains 125 supported requests, 25 missing-input requests, and 120 unsupported requests. It is never read by training. Artifact/corpus hashes are frozen in phase2-freeze.json. On this authored synthetic corpus, raw classifier macro F1 is 0.9711, the earlier rule baseline is 0.6854, and the typed router is 1.0000 with 0/120 unsupported requests accepted. The router adds deliberate policy and argument checks. These measurements describe this small corpus, not arbitrary language understanding or production accuracy. Wider independently collected user evaluation remains a Phase 5 gate.

## Workflows

- Read project-relative files, including “now read readme.md”; return actual bounded excerpts.
- Inspect up to 50 files and up to three supported root manifests; report manifest evidence and package.json identity/dependency names. Unknown project types are reported as unknown.
- Find filenames by supported glob; report incomplete coverage.
- Search literal code/text, rank returned matches by term count, and show exact paths, line numbers, and excerpts. Regex and whole-code explanation are outside this workflow.
- Run named test/build/lint checks from an explicit supported .nioxs.json configuration, Cargo mapping, or package.json scripts. Commands are returned to Nio, which enforces Build mode and approval. Arbitrary shell text from prompts is never executed.

## Persistence and permission checks

The [real smoke report](reports/phase2-smoke.json) records 32 passing checks using actual Nio, NioDB, and safe isolated local projects. Checks cover natural phrasing, manifests, file/code search, approved commands, Ask/Plan modes, denied approval, untrusted folders, no-tools/no-project-tools, missing inputs, configured command rejection, oversized results, concurrent continuation, provider restart, completed/uncertain outcomes, conflicting results, unexpected record revisions, cancellation, and database outages.

Immutable nano_runs/nano_actions transitions validate database revision 1, model version, request/action/result correlation, and logical workflow revisions. Idempotency keys coordinate claims; distinct provider owners cause conflicting concurrent command claims to fail. Retrying a dispatch with no outcome yields unknown_action; completed outcomes prevent replay even when output history is absent. Unavailable/conflicting storage stops dispatch. No exactly-once claim is made. Fresh requests receive unique turns; returned tool IDs retain the turn across continuation/restart. Explicit X-NioXS-Turn-ID is available to clients requiring initial-request deduplication.

## Limits

English/ASCII intent routing; project-relative paths without spaces, literal search only, bounded inventory and manifest excerpts, eight actions per workflow. No project edits, trained generative text, web workflow, MCP, or automatic LLM fallback. One deployment/operator per database. Records have 7-day expiry metadata, but cleanup is deferred to Phase 3; command claims must not be discarded in a way that permits replay. Record and journal scan caps fail closed. Hosted multi-tenant security and signed distribution remain later phases.

Reproduction: [development guide](../docs/DEVELOPMENT.md).
