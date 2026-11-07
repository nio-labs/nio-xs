# Manta CPU reference comparison — 2026-11-06

## Latest numerical isolation results

Found and fixed an actual IQ3 SIMD bug: sign-table entry 126 had a zero high byte instead of `+1`. The SIMD sign instruction therefore dropped one activation lane. The SIMD table is now generated from the canonical masks at compile time. A regression test checks all 128 sign patterns across seven SIMD kernel variants.

Also aligned Q8_0 activation quantization with GGML's CPU path (FP16-rounded scales, direct inverse scale, and ties-to-even rounding) and changed the RoPE angle cache to use GGML's F32 recurrence.

| Validation | Result after fixes |
| --- | ---: |
| Main dot products on actual Manta rows, 66 input fixtures | 264/264 exactly match GGML |
| Maximum specialized-kernel absolute error | 0.000003815 |
| Quantized activation bytes and scales | Exact match |
| Forced-prefix next-token choices, reference microbatch 1 / no weight repacking | 96/96 match |
| Forced-prefix next-token choices, default reference microbatch 512 / weight repacking | 94/96 match |
| Engine unit tests | 32 passed |

The original reference changes three of 96 token choices when switching from batched prefill to microbatch 1, even before turning weight repacking off. Exact logits are therefore not a portable invariant across these computation paths. Nonzero logit differences remain after the fixes; 96/96 agreement applies only to the tested prefixes and comparable reference configuration. Layer tracing shows close agreement at the first normalization and Q/K/V projections, with differences accumulating through attention and quantized layers.

See [numerical_isolation.json](numerical_isolation.json), [quant_kernel_comparison_fixed.json](quant_kernel_comparison_fixed.json), and [native_rope_aligned_forced.json](native_rope_aligned_forced.json) for the current results. The baseline and SiLU-only results below are retained for comparison. All changes remain local; no model weights were changed.

To reproduce the comparable reference path with the updated orchestrator, add `--reference-microbatch 1 --reference-no-repack` to the command below. The C++ driver also accepts a trailing trace JSONL path; the native runner accepts `--layers` to capture intermediate tensors.

## Original baseline comparison

Compared the same Manta GGUF and three prompts in the native engine and llama.cpp commit `6c01da6278ab6c69ed67d9124806b50d2b299ba7`. Native base revision: `790f9be`, with the local shared Manta prompts and filter changes. The model SHA-256 and exact configuration are recorded in `comparison.json`.

Both runtimes received identical token IDs. Greedy generation used no repetition penalty, suppressed tokens, speculative decoding, or output filtering. Both used F32 KV caches. Each free-running response was limited to 128 tokens. A second pass forced both runtimes through the first 32 reference tokens per prompt to compare next-token choices under identical prefixes.

| Prompt | Matching next-token choices before SiLU correction | After exact SiLU |
| --- | ---: | ---: |
| Rate limiter | 29/32 | 30/32 |
| Queue vs pub-sub | 29/32 | 30/32 |
| Deployment plan | 32/32 | 32/32 |
| Total | 90/96 | 92/96 |

The native SiLU implementation returned `x` for inputs >= 4 and zero for inputs <= -8. Replacing these approximations with `x / (1 + exp(-x))` improved argmax agreement on this sample, but did not establish numerical parity. Mean absolute differences on common top-10 logits did not consistently improve. See `silu_comparison.json` for the before/after metrics, and the raw trace files for logits.

The unfiltered reference responses also contain factual errors: the rate limiter decrements a counter after 100 requests without implementing a timed window, and the queue response claims retries prevent duplicate processing. The deployment response emits a Markdown fence despite the XML-only instruction and lacks dependency attributes in the observed steps. Responses reaching the 128-token cap cannot be used to diagnose spontaneous early stopping or full-document validity.

## Interpretation and limits

- Tokenizer mismatch was ruled out for these three inputs.
- Residual runtime differences require further isolation; they are not proof of a particular kernel bug.
- The same quantized artifact gives poor advice under the reference runtime too. This experiment cannot separate training problems from weight quantization effects.
- This is not a test of GPU execution, full 8k inputs, production repetition penalties, speculative decoding, or output filtering.
- No controlled speed comparison was performed. Removing the activation shortcut may affect throughput.
- Engine tests after the SiLU correction: 30 passed, zero failed.

## Reproduce

Build llama.cpp at the pinned commit, with its CPU backend. Build `scripts/reference/manta_runtime.cpp` against its `llama` library to produce a reference executable. The driver accepts explicit input and stop token ID files and checks its own tokenizer against the supplied input IDs.

Build the native runner:

```sh
cargo build --release -p nioxs-engine --example compare_manta_runtime
```

Run the Python orchestrator in an environment with `tokenizers` installed:

```sh
python scripts/compare_manta_runtime.py \
  --model "$HOME/.nioxs/models/manta.vmicrolm-vector.gguf" \
  --tokenizer "$HOME/.nioxs/models/manta-tokenizer.json" \
  --suite benchmarks/manta_reference_suite_2026-11-06.json \
  --reference /path/to/manta-reference \
  --reference-revision 6c01da6278ab6c69ed67d9124806b50d2b299ba7 \
  --output /tmp/manta-reference-rerun
```

The saved baseline traces predate the SiLU correction. A new run from the current working tree uses exact SiLU. The isolated after-correction forced traces are in `native_exact_silu_forced.json`.
