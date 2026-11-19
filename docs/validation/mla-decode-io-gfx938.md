> Current W4A8 block64 validation passed exact native consumers and changing
> graphs. The combined Q/RoPE + MLA + draft Post stack passed full159/164,
> first32=32 and paired serving performance/8-rank trace gates. See
> `w4a8-mla-reuse.md`. Isolated timings below belong to the earlier FP8
> experiment and are not W4A8 performance claims.

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

At M48/H64, complete-bridge graph timing is 151.23 -> 117.09 us for SWA-only
and 224.26 -> 185.13 us for combined attention. These are isolated timings;
HumanEval and formal serving admission remain pending.

Run CPU checks with HCU_ADAPTER_HOST=none and the three tests in
`tests/adapter/test_mla_decode_io.py`, `test_mla_selectors_contract.py` and
`test_vllm_flashmla_decode.py`. Run GPU validation with
`python tests/tools/probe_mla_decode_io.py --baseline
<accepted-759bb3f>/hcu_adapter/kernels/mla_sparse_decode_dtk.py --output <json>`.
Raw artifacts: `/public2/home/duyunlong/run/dsv4.1/033/moe_reverify_20261118/mla_io_*`.
