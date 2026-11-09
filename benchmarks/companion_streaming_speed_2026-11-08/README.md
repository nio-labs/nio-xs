# Skate and Manta visible streaming speed

Measured 2026-11-08 on Intel Core i7-8650U (4 cores / 8 threads), WSL2,
using the optimized native CPU engine and default per-model settings.
Build: NioXS 0.5.2 with the local shared companion prompt changes.

Prompt for both models: `Explain how a refrigerator moves heat in about 120 words.`
Each model is loaded once and receives the same request twice. Output budget:
256 model tokens per request, matching the preceding Beluga streaming probe.

| Model | Request | Output words | Words after first chunk | First-to-last chunk time | Streaming words/s | Wait to first chunk after loading |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Skate | 1 | 28 | 24 | 0.339 s | 70.69 | 0.883 s |
| Skate | 2 | 28 | 24 | 0.297 s | 80.72 | 0.649 s |
| Manta | 1 | 171 | 167 | 7.204 s | 23.18 | 3.235 s |
| Manta | 2 | 171 | 167 | 6.856 s | 24.36 | 2.420 s |

Loading is separate: 0.365 s for Skate and 1.634 s for Manta. Streaming rates
exclude loading, prompt processing, the words already present in the first
chunk, and shutdown. They use subsequent visible words divided by the time from
the first nonempty output chunk to the last. Displayed thought text counts as
output; think delimiters and standalone Markdown markers do not count as words.

These are application-visible chunk rates, not exact model-token decode rates.
Skate's short output makes its rates especially sensitive to buffering and
scheduling. The models produced different lengths despite the same prompt;
this is not a controlled comparison of model efficiency or answer quality.
Responses and complete timing data are in `skate.jsonl` and `manta.jsonl`.

Reproduce from the repository root, with other inference and compilation jobs
stopped:

```sh
cargo build --release -p nioxs-engine --example bench_beluga_latency
target/release/examples/bench_beluga_latency --model skate > /tmp/skate-streaming.jsonl
target/release/examples/bench_beluga_latency --model manta > /tmp/manta-streaming.jsonl
```

The example defaults to Beluga for compatibility with its original invocation;
`--model` selects the companion whose production engine and system prompt run.
