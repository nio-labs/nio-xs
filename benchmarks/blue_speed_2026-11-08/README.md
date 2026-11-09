# Blue latency investigation — 2026-11-08

Host: Intel Core i7-8650U, four physical cores, WSL2. Native CPU,
optimized NioXS 0.5.2 engine, default Blue layer-residency settings.
Question: `Explain how a refrigerator moves heat in about 120 words.`
Each process loads Blue once, then makes two sequential requests with a
256-token budget. No compilation or other benchmark inference overlapped runs.

## Kept change: shorter shared Blue system prompt

The shared prompt retains direct answers, accuracy, format compliance,
concision, relevance, standard terminology, avoiding invented APIs/details,
and stating uncertainty. The CLI, API, and raw engine calls share this policy.

`baseline_original` uses the saved pre-change optimized benchmark binary and
the original policy. `baseline_short_prompt` uses that same binary with the
new policy explicitly framed as ChatML, so all inference kernels are unchanged.
`candidate_short_prompt` uses the same new policy and a trial kernel change.

| Configuration | First buffered output after loading, requests 1 / 2 | Visible streaming words/s, requests 1 / 2 | Output words | Sampled peak anonymous RAM |
| --- | ---: | ---: | ---: | ---: |
| Original policy, original kernel | 18.09 / 14.04 s | 2.37 / 2.29 | 33 each | 193.27 MiB |
| Short policy, original kernel | 10.16 / 10.76 s | 2.68 / 2.46 | 53 each | 184.14 MiB |
| Short policy, trial paired kernel (reverted) | 10.39 / 10.37 s | 2.51 / 2.46 | 53 each | 184.14 MiB |

First-output delay excluding loading fell about 23–44% in these observations.
Loading took 0.60 s for the original-policy process and 0.19 s for the
short-policy baseline. Including loading, the first request's initial wait fell
from 18.69 s to 10.35 s. Filesystem caches were not cleared; these are small
host-specific probes, not guaranteed latency bounds.

Streaming excludes loading, prefill, the first chunk's words, and shutdown.
It is visible words after the first chunk divided by first-to-last nonempty
chunk time. Chunk buffering means this is not exact model-token decode speed.
Different policy outputs have different lengths and wording, so their word
rates do not establish a kernel throughput improvement. Both shortened-policy
variants produced identical output in both requests. All responses still need
factual review; the refrigerator explanation contains an incorrect temperature
claim. This work does not establish answer quality or production readiness.

## Reverted experiment

The trial fed gate and up rows into the existing paired-row / three-token
IQ3 kernel together, instead of making separate three-token dot-product calls.
Numerical checks passed for 256-, 3072-, and 8192-column matrices with distinct
inputs and one/four worker partitions; existing IQ3 sign-pattern tests passed.
The matched-prompt measurements showed no useful speed gain. The kernel change
and its experiment-specific test were removed. Original inference math and
residency defaults remain in use.

## Other findings

- Blue evaluates all 32 transformer layers for confirmed output tokens. Its
  memory-saving layer streaming does not remove that computation.
- The exported GGUF advertises a Phi-style template, but the local Blue training
  script explicitly trains ChatML with aliased special tokens. The runtime's
  ChatML format was retained. Future reference comparisons must supply the
  actual training format rather than blindly trusting the stale template.
- A CPU sample profile was collected, but this host's perf build lacks ELF/DWARF
  symbol support and the release binary is stripped. Samples cannot identify
  individual kernels from that profile. No function-level attribution is claimed.
- These results justify reducing prompt overhead. They do not meet a high-speed
  Reflex target; further work needs effective batched prefill and faster
  quantized projection kernels, with numerical and performance verification.

Raw responses, phase timings, binary hashes, and 10 ms sampled RAM peaks are
in `comparison.json` and the associated JSONL files. Sampling may miss brief
memory peaks. This patch changes the shared Blue prompt only; the trial math
optimization was discarded. It does not install a GPU backend or change weights.

## Final rebuilt source validation

After restoring the original kernel, the optimized provider CLI and latency
example rebuilt successfully. The shared prompt regression passed. Two final
engine requests in `final_validation.jsonl` matched the shortened-policy baseline
responses exactly (53 words each):

| Request | First buffered output after loading | Streaming words/s |
| --- | ---: | ---: |
| 1 | 11.32 s | 2.70 |
| 2 | 9.76 s | 2.58 |

Model loading took 0.274 s, making the first request's load-plus-first-output
delay 11.59 s. These confirm lower prompt overhead, with host variability; they
do not demonstrate a large improvement in sustained token computation.
