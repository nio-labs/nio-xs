# Beluga CPU speed check — 2026-11-08

Target: at least 10 visible words/s, preferably 15. **Not met on this host.**

Hardware: Intel Core i7-8650U, 4 physical cores / 8 logical CPUs, WSL2,
approximately 3.8 GiB available VM RAM. Optimized NioXS 0.5.2 binary;
its SHA-256 is recorded in `results.json`.

Model: manifest Beluga v0.3.0, 639,362,560 bytes,
SHA-256 `d2d4642d36b907c8658301208e8cd4ab3c6d72051a37a780ebc35a56ff83d5ba`.
Tokenizer SHA-256:
`cf535eea1ed9374c2bc3436625216c18fc703fc37aae8e019b3e85cc658e0395`.

## CLI measurements

Each run starts a new process, forces native CPU execution, and includes startup,
prompt processing, generation, and shutdown. Two identical prompts are tested
across layer settings with a 96-token output cap. Visible whitespace-separated
words must contain an alphanumeric character; standalone Markdown markers and
think delimiters do not count. Displayed thought text counts as output.

| Resident layers / rolling window | Words/s, aggregate | Sampled peak anonymous RAM | Sampled peak resident RAM |
| --- | ---: | ---: | ---: |
| 0 / 1 (default) | 3.66 | 120–124 MiB | 276–277 MiB |
| 14 / 1 | 3.80 | 122–124 MiB | 515–516 MiB |

Aggregates use total words divided by total elapsed time for the first six runs
(three per setting). The last two runs may overlap a benchmark compilation and
are marked `compilation_overlap_possible` and excluded. All output text matched
between layer settings for each prompt. This checks output equivalence for these
probes, not factual accuracy or general determinism.

The measured gain is approximately 4%, with about 239 MiB more mapped resident
RAM. Default memory settings were not changed. An earlier single probe with all
28 layers resident reached 4.01 words/s and about 721 MiB RSS; this exploratory
result is not used in the aggregate comparison. Its prompt was the sky question.

## Loaded model reuse

`reused_engine.jsonl` measures two identical requests using one production
`AstEngine`, with the low-memory defaults. The prompt asks for about 120 words;
the model produced 143 counted words each time, with a 256-token output cap.

- Model/tokenizer loading: 0.624 s.
- First request including prompt processing: 29.815 s, 4.80 words/s.
- First request including loading: 4.70 words/s.
- Second request including prompt processing: 29.331 s, 4.88 words/s.
- First buffered output: 5.69 s and 5.36 s respectively.

Request rates include prefill and must not be advertised as decode-only rates.
First chunk timing observes production buffering, not exact first-token timing.
Model reuse and increased residency did not reach the requested target.

## Visible streaming rate, excluding startup and prefill

`streaming_speed.jsonl` records a separate run with the benchmark extended to
timestamp the first and last nonempty chunks. Each response contains 143 words;
the first chunk contains four words, excluded from the throughput numerator.

| Request | Wait to first chunk after loading | First-to-last chunk duration | Words after first chunk | Streaming words/s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.81 s | 22.38 s | 139 | 6.21 |
| 2 | 4.90 s | 22.63 s | 139 | 6.14 |

Model loading took 0.753 s separately. Streaming rates exclude loading, prefill,
the first chunk's words, and shutdown. They measure application-visible output;
production chunk buffering means they are not exact model-token decode rates.
The 10–15 words/s goal remains unmet using this measurement too.

## Reproduce

Run from the repository root with other compilation/inference jobs stopped:

```sh
cargo build --release -p nioxs-provider
python3 scripts/benchmark_beluga_speed.py --output /tmp/beluga-speed.json
cargo build --release -p nioxs-engine --example bench_beluga_latency
target/release/examples/bench_beluga_latency > /tmp/beluga-reuse.jsonl
```

Linux RSS/anonymous memory is sampled every 10 ms; brief peaks can be missed.
Filesystem caches are not cleared. This is a small host-specific performance
probe, with no claim about response quality or production readiness.

`cpu_profile.txt` contains an exploratory 99 Hz user CPU sample report. The local
perf tool did not resolve function symbols, so it cannot identify the hot kernel.
Further CPU optimization needs a symbolized profile of prefill and generation,
followed by output-equivalence and RAM checks. No decoding shortcuts, model
changes, or speculative-acceptance changes were made for this speed check.
