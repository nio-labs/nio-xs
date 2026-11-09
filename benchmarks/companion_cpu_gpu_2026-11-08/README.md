# Companion CPU and GPU measurements — 2026-11-08

Host: Intel Core i7-8650U (4 cores / 8 threads), WSL2. Optimized NioXS 0.5.2 native engine with local shared-prompt changes.

## CPU

| Model | Model/tokenizer load | First response including load and prefill | Streaming words/s, two requests | Output words per request |
| --- | ---: | ---: | ---: | ---: |
| Skate | 0.17 s | 0.85 s | 78.97–90.60 | 28/28 |
| Manta | 0.80 s | 3.08 s | 24.10–24.27 | 171/171 |
| Beluga | 0.71 s | 6.99 s | 6.12–6.24 | 143/143 |
| Blue | 0.88 s | 21.49 s | 2.16–2.27 | 33/33 |

## GPU

| Model | Startup | Streaming words/s | Status |
| --- | ---: | ---: | --- |
| Skate | — | — | Unavailable |
| Manta | — | — | Unavailable |
| Beluga | — | — | Unavailable |
| Blue | — | — | Unavailable |

No llama-cli on PATH and neither NIOXS_GPU_RUNNER nor NIOXS_LLAMA_CLI is configured. GPU device nodes are visible but no GPU execution was measured.

GPU device visibility alone does not establish that an inference backend works. No CPU fallback is reported as a GPU run. Installing or rebuilding GPU runtimes was not part of this measurement.

## Method

- Same prompt for all models: `Explain how a refrigerator moves heat in about 120 words.`
- 256-token output budget for each request; two sequential requests using one loaded model per companion.
- Models run sequentially, with no overlapping builds or other benchmark inference.
- Model/tokenizer loading is measured once per companion, after the benchmark runtime starts. OS process launch and CLI routing are excluded.
- First response includes model loading plus prompt processing and buffering before the first nonempty output chunk. It is a first-request observation, not an average or a cold-disk test. Filesystem caches were not cleared.
- Streaming rate = visible words after the first chunk divided by time from the first to the last nonempty chunk. Startup, prefill, first-chunk words and shutdown are excluded.
- Production phrase buffering makes this application-visible streaming speed, not exact model-token decode speed.
- Counted words are whitespace-separated items containing an alphanumeric character; think delimiters and standalone Markdown markers are excluded. Displayed thought text counts if present.
- Skate produced only 28 words, and Blue 33, making their short streaming intervals especially sensitive to buffering/scheduling. Different response lengths limit cross-model comparison.
- These runs measure speed only. Their answers contain errors; no production-readiness or factual-quality claim is made.

## Reproduce CPU measurements

```sh
cargo build --release -p nioxs-engine --example bench_beluga_latency
NIOXS_DEVICE=cpu target/release/examples/bench_beluga_latency --model skate
NIOXS_DEVICE=cpu target/release/examples/bench_beluga_latency --model manta
NIOXS_DEVICE=cpu target/release/examples/bench_beluga_latency --model beluga
NIOXS_DEVICE=cpu target/release/examples/bench_beluga_latency --model blue
```

Run from the repository root with the default memory settings. Raw responses and timings are in the model JSONL files. `summary.json` records the binary SHA-256 and report metadata; `comparison.csv` includes separate model-load and first-response columns, with blank GPU values.
