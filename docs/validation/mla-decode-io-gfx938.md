# Exact native MLA decode input/output scheduling on gfx938

Full H64/H128 contiguous, 256-byte-aligned native queries/sinks can be passed
through without allocating zero-filled copies. Partial heads, unaligned or
strided sources keep the original padding/copy path. This admission is bounded
to gfx938 and 1..64 rows. The native attention provider, fresh scheduling plan,
selector values, cache layout and inverse-RoPE contract remain unchanged.

A guarded BF16 output kernel combines the native result copy and empty-row
mask. It reads current main/extra counts on every invocation and copies BF16
bits unchanged for live rows, or positive zero for empty rows. Non-contiguous
outputs, partial heads, other architectures and MXFP8 output retain the
original postprocessing.

45 CPU checks pass. The GPU probe passes 160 changed-input mask cases,
including all BF16 bit patterns, and 1,152 full native decode cases in both
eager and dynamic Graph replay. Inputs are immutable and outputs byte-exact
against accepted bridge 759bb3f. Coverage: all eight capture buckets,
H4/H8/H64/H128, SWA-only and combined paths, contiguous/strided/misaligned
queries, sinks and outputs, magnitudes 0/.001/.25/1/8/100 and changing Q,
cache, positions, sinks and main/extra selectors. Native ABI is the installed
FlashMLA 1.2.0+dtk2610 wheel. No plan caching is introduced.

At M48/H64, final-source complete-bridge graph timing is 151.005 -> 117.906 us
for SWA-only and 219.794 -> 185.072 us for combined attention. Both final
CPU and GPU checks pass; production source is 03ef92e.

HumanEval first32 passes 32/32 with identical baseline tasks/prompts, zero
errors/truncations and real prefill/acceptance. Both unprofiled 64-request
repeats complete with zero failures: median TPOT 22.223018363 / 21.874368957 ms,
mean of repeat medians 22.048693660 ms. All frozen matched/aggregate and prior
SLA gates pass. This is 2.956% below the accepted selector control and 40.437%
below the original baseline, based on two repeats without a confidence claim.

The separate profile and workflow exit zero with eight traces. Each rank
selects 400 target and 30 draft mask/copy kernels per ten steps; target mean
8.217..8.361 us, draft 6.532..6.731 us. Existing target Q/KV/Post and native
draft Q/KV paths remain selected. Phase medians: target 82.611..82.812 ms,
draft/sampling 6.755..6.902 ms. Median GPU idle fractions are 0.881..0.922%
and 0.746..0.780%. Full SGLang stage parity remains incomplete.

Measured plugin is ec93021 (production model code a4687f8). Subsequent retained
plugin test CRLF/module EOF normalization has an identical Python AST.

Run CPU checks with HCU_ADAPTER_HOST=none and the three tests in
`tests/adapter/test_mla_decode_io.py`, `test_mla_selectors_contract.py` and
`test_vllm_flashmla_decode.py`. Run GPU validation with
`python tests/tools/probe_mla_decode_io.py --baseline
<accepted-759bb3f>/hcu_adapter/kernels/mla_sparse_decode_dtk.py --output <json>`.
Raw artifacts: `/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/mla_io_*`.
