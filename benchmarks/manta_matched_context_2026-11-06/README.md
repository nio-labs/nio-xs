# Manta context retrieval comparison (2026-11-06)

All fixtures use the same exact tokenizer ID construction as `bench_context_capacity`: system prompt, the BRONZE marker near the beginning, SILVER around the midpoint, GOLD near the end, and repeated archive filler. llama.cpp was given those explicit IDs (no re-tokenization), CPU only, greedy decoding, F32 KV, 512-token microbatches, and 32 output tokens. This is a synthetic marker-retrieval probe, not a broad long-context benchmark.

| Prompt tokens | Runtime | Result |
|---:|---|---|
| 512 | NioXS | Retrieved BRONZE, SILVER, GOLD |
| 512 | llama.cpp | Retrieved BRONZE, SILVER, GOLD |
| 1,024 | NioXS | Retrieved BRONZE, SILVER, GOLD; prefill 22.20 s |
| 1,024 | llama.cpp | Retrieved BRONZE, SILVER, GOLD |
| 2,048 | llama.cpp | Failed retrieval; repeated “The service is healthy.” |
| 8,160 | NioXS | Failed retrieval; repeated filler; prefill 604.39 s |

The 4,096-token llama.cpp run was interrupted after about 10 minutes without producing a response. The 6,144- and 8,160-token llama.cpp comparisons were therefore not completed. No inference should be drawn about whether llama.cpp succeeds or fails at 8k from this run. The 2k reference failure does show that this fixture/model combination has a retrieval-quality problem independent of NioXS at or below that length.

The GGUF metadata advertises context length 32,768 and RoPE base 1,000,000. Local release notes describe a 2,449-turn fine-tuning corpus but do not state its sequence-length distribution or maximum. Therefore the training context length remains unverified. Context capacity and correct rejection/recovery at 8,192 were separately verified in `benchmarks/manta_release_checks_2026-11-06/context_8k.json`; those checks do not establish 8k retrieval quality.

## Files

- `input_*.txt`: exact prompt token IDs used by llama.cpp.
- `reference_*.json`: completed llama.cpp responses and generated IDs.
- `native_1024.json`: native 1k result and timing.
- `reference_4096.stderr.txt`: partial diagnostics from the interrupted run.
