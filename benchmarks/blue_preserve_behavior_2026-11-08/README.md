# Blue behavior-preserving performance trial

Host: Intel Core i7-8650U, four physical cores / eight threads, WSL2.
Optimized native CPU engine. Blue's model weights, tokenizer, system prompt,
greedy decoding, repetition penalty, token budget and memory settings stayed
fixed throughout this trial. `comparison.json` records the model, benchmark
binary and system-prompt SHA-256 values.

## Results

All **10 measured runs** across three prompts produced token sequences identical
to their controls. The six refrigerator runs also matched the trace captured
before adding the scheduling controls. This is measured equivalence for these
cases, not a universal proof that floating-point execution paths always agree,
or evidence that the original answers are factually correct.

| Refrigerator mode | Streaming words/s, aggregate | Mean decode time | Sampled peak anonymous RAM |
| --- | ---: | ---: | ---: |
| `on` — current default | 2.223 | 23.723 s | 184.14 MiB |
| `adaptive` — experimental backoff | 2.305 | 22.963 s | 184.14 MiB |
| `off` — autoregressive control | 2.255 | 23.717 s | 183.70 MiB |

The approximately 3.7% visible-rate gain for adaptive mode is small and needs
broader measurement. HTTP 429 and arithmetic checks showed no meaningful
performance gain. The default therefore remains `on`; no kernel alternative
was enabled. This trial does not meet a 10–15 words/s target.

The default refrigerator run proposed 22 draft tokens and accepted six (27.3%).
Adaptive scheduling reduced proposals to 20 while still accepting six. It
requires two unsuccessful drafting steps before using eight autoregressive
steps, then probes drafting again. Successful verification resets the miss
streak. Every accepted draft still passes the original full-model verifier.

The other test prompts were a one-sentence explanation of HTTP 429 and
`What is 17 + 25? Answer only with the number.` Their output sequences contain
21 and two tokens respectively, with arithmetic returning `42`.

## Timing and equivalence method

- Two rounds of the refrigerator prompt, alternating mode order (`on`,
  `adaptive`, `off`, then the reverse), with one request per process.
- Same prompt as previous speed reports, asking for about 120 words; 96-token
  budget. All modes completed the same 69-token / 53-word response.
- HTTP 429 and arithmetic use a 64-token budget and one run per mode (`on`
  versus `adaptive`).
- Streaming words/s counts words after the first nonempty output chunk divided
  by time from first to last chunk. Loading, prompt processing, first-chunk
  words and shutdown are excluded; chunk buffering makes this an application
  streaming metric, not an exact token-level metric.
- Diagnostic decode time begins after prefill and includes sampling, forward
  passes, verification and streaming. Reported generated tokens exclude stop
  tokens. Generated token IDs are compared directly, rather than retokenizing
  the visible answer.
- RAM is sampled from the child process every 10 ms; brief peaks can be missed.
- No compilation or other benchmark inference overlapped measurements.
- First-request latency and rates vary with host/cache state. Compare modes
  within this trial rather than treating differences from earlier reports as
  gains or regressions.

## Decode profile

`decode_profile.json` samples user CPU activity after the first visible output.
A profiling CLI was built with symbols retained (`-C strip=none`), with the same
optimization and inference settings. Because this host's perf lacks libelf,
sample file offsets were converted through the matching ELF `PT_LOAD` segments
and resolved against its function symbol table. 8,136 / 8,187 samples resolved.
The matching binary hash is recorded in the profile JSON.

Approximate sampled CPU activity:

| Function/group | Share |
| --- | ---: |
| IQ3 two-row dot-product kernel | 59.9% |
| Worker pool closure, including waits/spinning | 19.0% |
| IQ3 three-token dot-product kernel | 12.9% |

CPU sample percentages across threads are not wall-time percentages. This
identifies the two-row quantized arithmetic as the main optimization candidate;
it does not predict a particular end-to-end speedup.

## Rejected arithmetic experiments

The existing four-row IQ3 kernel produced bit-identical results on synthetic
Blue-sized matrices, but was slower than the existing two-row kernel. The
isolated medians were 5.54 versus 6.82 ms (8192 × 3072) and 5.75 versus 6.76 ms
(3072 × 8192). No production call site was changed.

A prototype replacing scalar codebook loads with AVX2 gathers also produced
bit-identical matrix results, but showed negligible gains in the short probe:
5.82 versus 5.80 ms and 5.65 versus 5.53 ms. The prototype was removed. Its raw
measurements remain in `gather_microbenchmark.jsonl`.

## Run the opt-in trial locally

```sh
cargo build --release -p nioxs-provider -p nioxs-engine --bin nioxs --example bench_beluga_latency
NIOXS_DEVICE=cpu NIOXS_INFERENCE_STATS=1 NIOXS_BLUE_SPECULATION=on target/release/examples/bench_beluga_latency --model blue --max-tokens 96 --repeat 1
NIOXS_DEVICE=cpu NIOXS_INFERENCE_STATS=1 NIOXS_BLUE_SPECULATION=adaptive target/release/examples/bench_beluga_latency --model blue --max-tokens 96 --repeat 1
NIOXS_DEVICE=cpu NIOXS_INFERENCE_STATS=1 NIOXS_BLUE_SPECULATION=off target/release/examples/bench_beluga_latency --model blue --max-tokens 96 --repeat 1
```

`NIOXS_BLUE_SPECULATION` controls Blue's native path only. Accepted values are
`on` (default), `adaptive`, and `off`; invalid values fail explicitly. Other
companions retain their existing scheduling. `NIOXS_INFERENCE_STATS=1` emits
phase timings, counters and generated token IDs to stderr on normal completion.
It is off by default. These are local diagnostic controls, with no release or
installation performed by this trial.

Validation: three schedule tests passed; the optimized engine example and
provider CLI built successfully. Future optimization should target the measured
IQ3 hot path and continue comparing numeric results and full token traces.
