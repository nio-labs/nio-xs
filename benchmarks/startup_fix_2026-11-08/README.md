# Startup delay fix — 2026-11-08

The installed `nioxs` initially resolved to an older optimized binary at
`/home/mn/.local/bin/nioxs`. This investigation used a saved copy of the current
pre-fix release binary to isolate the backend-selection change from the older
installed build and previous prompt changes.

## Changes

- Resolve GPU runner availability before probing hardware in the model path.
  With no runner present, native CPU inference uses CPU residency settings.
- Reuse one Windows adapter inventory per process rather than issuing the same
  PowerShell CIM query repeatedly.
- Make the PowerShell query noninteractive and cap it at three seconds.
  NVIDIA utility and Vulkan summary probes have two-second limits.
- Use the shared adapter query for native Windows Arc discovery as well.

Weights, prompts, greedy decoding and the default speculation mode were held
fixed in the before/after comparison. No GPU runtime was installed.

## Checks

Three focused tests passed: adapter inventory parsing, completed command output,
and termination/reaping of a stalled probe. The optimized provider built.

A separate fixture supplies a mock PowerShell adapter inventory and an absent
GPU runner. The pre-fix engine queried the inventory twice; the fixed engine
queried it zero times. Both returned the same Skate text. This verifies fallback
control flow, not GPU execution. See `probe_regression.json`.

Live Blue probe on Intel Core i7-8650U / WSL2, automatic device preference:

`What is 17 + 25? Answer only with the number.`

| Build | Launch to first visible output | Native prefill | Native decode | Answer |
| --- | ---: | ---: | ---: | --- |
| Before | 19.50 s | 14.52 s | 0.93 s | 42 |
| After | 14.08 s | 12.48 s | 1.03 s | 42 |

Generated token IDs match exactly. First visible output was sampled from the
CLI output file every 10 ms and includes process launch, backend selection,
loading, prefill, sampling and output buffering. This single before/after probe
is about 28% faster to first output, but cache/host variability and corrected
residency also affect the timings. It is not a guaranteed speed bound or a
streaming-throughput benchmark. The original CIM query independently took
1.36 s on this host.

The remaining roughly 12.5-second wait is native model prefill in this probe;
the change does not eliminate Blue's CPU computation cost or establish model
answer quality. Raw output, binary hashes, counters and timings are stored in
`blue_comparison.json` and its associated files.

## Local installation

The validated release binary was installed atomically at
`/home/mn/.local/bin/nioxs`, with the prior binary preserved at the path recorded
in `installation.json`. Installed and built binary SHA-256 values match, and
`nioxs --version` returns `0.5.2`. This is a local working-tree build; no Git
release, tag or push was performed.
