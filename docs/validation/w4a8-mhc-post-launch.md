# W4A8 mHC Post launch policy

The admitted H5120/HC4 gfx938 Post kernel launched too many small CTAs at
decode sizes. Keep its arithmetic and existing admission/fallbacks, and select
BLOCK1024/2 warps for32..64 rows, BLOCK256/8 warps for1..16 rows, and the previous
BLOCK256/4 warps otherwise. No serving option or environment variable is added.

The unchanged kernel preserves the official FP32 accumulation order, separate
post product/addition and BF16 storage boundary. CUDA autocast and unsupported
precision/layout/device contracts retain the original implementation.

`tools/probe_mhc_post_launch.py` compares12 block/warp combinations against the
actual native Torch Post. On the fixed gfx938/64CU W4A8 environment,720 eager
comparisons and480 changing-input graph replays are byte-exact, including rows
through1024. Three shuffled graph benchmark repeats give rows48 median
7.311046 ->4.296037us and rows64 median9.386778 ->5.407016us. These are operator
timings, not whole-service speedup claims.

The complete service with Plugin a5bc072d0a1cb80f97f1a69db771a09525f79d09 and
tested Adapter18c695303a0548b134a39a1a34c4f0ce4b472391 passes real-prefill
HumanEval159/164 (96.95%), first32=32, zero errors/missing/truncations. Both
formal64-request repeats complete without failures. The same source/runtime
packages and serving/workload arguments are used on nmz18:

| Client TPOT | Prior Post launch | Candidate |
| --- | ---: | ---: |
| Repeat1 median, ms |20.122414|19.923699|
| Repeat2 median, ms |19.570560|19.587913|
| Repeat1 p99, ms |23.792888|23.462990|
| Repeat2 p99, ms |24.297764|24.505892|
| Mean of medians, ms |19.846487|19.755806|

The frozen per-repeat median2%/p995% regression ceilings and mean2% ceiling
pass. The0.4569% mean difference does not meet the separate2% per-repeat
end-to-end improvement threshold; no such claim is made. All8 independent
traces verify a single FULL target graph,40 target/3 draft LL MoE and activation
calls, and at least83 target/6 draft Post calls per measured phase.

Performance uses fake zero KV, five literal unconditional0.6 acceptance rates,
DSpark5,4k input/128 output and64 requests/concurrency. Complete accuracy uses
real prefill and real rejection sampling. TP1/DP8/EP8,0.85 memory,block64,seq32,
len32768,batched256,capture1/4/8/16/32/40/48/64 and FULL_AND_PIECEWISE remain fixed.

DP0 target-to-following-draft-end median changes80.861678 ->80.683877ms; the
supplied SGLang reference remains65.098274ms on a different hardware/batch/
provider context. Whole-step alignment is unfinished. See the adjacent JSON
for artifact hashes, exact sources and validation limits.
