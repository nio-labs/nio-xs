# Manta release checks — 2026-11-06

## Implemented

- Explicit CLI `manta --plan` mode buffers output and validates one complete XML document, required step fields, unique IDs, dependency references, and acyclicity. Invalid plans are withheld and return a nonzero status. The ONNX fallback uses the same validation gate. Commands are not executed or checked for semantic correctness.
- Native and external-runner settings use one greedy policy: repetition window 64, frequency/presence penalties zero, shared per-model repetition penalties and suppressed token IDs. Native history now includes the prompt. Legacy frequency subtraction, repeated exponentiation, a hidden 96-token window, and the `1.0`-to-`1.25` penalty override were removed.
- Oversized prompts fail before generation. The budget respects the last predictable context position; the last output can be predicted without being forwarded into the KV cache.

Temperature/top-p arguments remain accepted for API compatibility, but this native companion path is greedy; GPU configuration now reflects that behavior too. No weights or training data were changed.

## Verification

37 engine tests and 78 provider tests pass. The tests include positive/negative XML cases, graph validation, repeated-token scoring on scalar and SIMD vocabulary kernels at companion dimensions, context budgets, and shared runner arguments.

GPU execution was not measured. The environment exposes graphics device nodes bound to `faux_driver` and Windows driver DLLs under WSL, but no usable Linux GPU runner/runtime was available. See [gpu_status.json](gpu_status.json).

## Small 0.5B-scope tasks

| Task | Manual quality review | Runtime/protocol result |
| --- | --- | --- |
| HTTP 429 explanation | Incorrect: fails to identify rate limiting / Too Many Requests | Completed without runtime error |
| Work queue / worker explanation | Correct at the requested introductory level | Completed without runtime error |
| Idempotency key explanation | Incorrect: confuses deduplication with preventing retries | Completed without runtime error |
| Three-step stateless API rollout | Partial: three steps, but staging explanation is weak | Completed without code fences |
| One-step XML plan | Does not satisfy the requested schema | Withheld; exit status 1 |
| Two-step XML plan | Missing dependency attribute | Withheld; exit status 1 |

These six deterministic checks are not a general capability benchmark. Model quality has not passed a release gate merely because runtime and validator tests pass. Full prompts, responses, stderr, timing and sampled anonymous memory are in [short_tasks.json](short_tasks.json).

## Context probe

The native probe uses exact token counts assembled from the Manta tokenizer. It places distinct markers near the beginning, middle, and end of repetitive filler, then asks for the three names. It measures full prefill and autoregressive decoding separately, samples `RssAnon` / `VmRSS` every 10 ms, rejects oversized prefill and context overruns, and checks recovery after resetting the cache. Brief memory peaks can be missed. This synthetic test does not establish general long-context answer quality or GPU behavior.

| Prompt tokens | Prefill | Decode | Sampled peak anonymous RAM | Sampled peak resident RAM | Marker retrieval |
| ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 8.82 s | 0.89 s | 141.87 MiB | 286.21 MiB | All three markers returned |
| 8,160 | 604.39 s | 4.85 s | 312.39 MiB | 497.73 MiB | Failed; repeated filler text |

Both runs rejected oversized input and context overruns, then successfully prefetched a fresh short prompt and generated its first token after resetting the cache. The long run exercised all 8,192 cache positions. These boundary/recovery checks passed; the 8k retrieval check did not.

The 512-token control returned “The START marker is BRONZE, the MIDDLE marker is SILVER, and the END marker is GOLD.” At 8,160 tokens, the response repeated “The service is healthy.” until reaching the 32-token output budget. This is evidence of failure on the synthetic retrieval fixture, not proof about every long-context workload. The probe uses unfiltered greedy decoding with repetition penalties disabled; the companion CLI uses its configured repetition penalty.

See [context_short.json](context_short.json) and [context_8k.json](context_8k.json) for full records. The final measured runs were sequential, after the short-task suite, without concurrent compilation. An earlier prefill was interrupted after discovering the legacy penalty override and was excluded from the measurements.

## Reproduce

```sh
cargo test -p nioxs-engine --lib
cargo test -p nioxs-provider --lib
cargo build --release -p nioxs-provider --bin nioxs
python3 scripts/check_manta_release.py
cargo build --release -p nioxs-engine --example bench_context_capacity
cargo run --release -p nioxs-engine --example bench_context_capacity -- \
  "$HOME/.nioxs/models/manta.vmicrolm-vector.gguf" \
  "$HOME/.nioxs/models/manta-tokenizer.json" 8160
```

The full context probe can take minutes on CPU. Its default 8,160-token prompt leaves room for up to 32 output tokens and then fills the cache to exercise the 8,192-token boundary.
